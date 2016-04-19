#     function bar!(jobz::Char, range::Char, dv::Vector{$elty}, ev::Vector{$elty}, vl::Real, vu::Real, il::Integer, iu::Integer)
#         n = length(dv)
#         if length(ev) != n - 1
#             throw(DimensionMismatch("ev has length $(length(ev)) but needs one less than dv's length, $n)"))
#         end
#         eev = [ev; zero($elty)]
#         abstol = Array($elty, 1)
#         m = Array(BlasInt, 1)
#         w = similar(dv, $elty, n)
#         ldz = jobz == 'N' ? 1 : n
#         Z = similar(dv, $elty, ldz, n)
#         isuppz = similar(dv, BlasInt, 2n)
#         work = Array($elty, 1)
#         lwork = BlasInt(-1)
#         iwork = Array(BlasInt, 1)
#         liwork = BlasInt(-1)
#         info = Ref{BlasInt}()
#         for i = 1:2
#             ccall((@blasfunc($stegr), liblapack), Void,
#                 (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty},
#                 Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
#                 Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
#                 Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
#                 Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
#                 &jobz, &range, &n, dv,
#                 eev, &vl, &vu, &il,
#                 &iu, abstol, m, w,
#                 Z, &ldz, isuppz, work,
#                 &lwork, iwork, &liwork, info)
#             chklapackerror(info[])
#             if i == 1
#                 lwork = BlasInt(work[1])
#                 work = Array($elty, lwork)
#                 liwork = iwork[1]
#                 iwork = Array(BlasInt, liwork)
#             end
#         end
#         w[1:m[1]], Z[:,1:m[1]]
#     end
# end
# bar!(jobz::Char, dv::Vector, ev::Vector) = bar!(jobz, 'A', dv, ev, 0.0, 0.0, 0, 0)
